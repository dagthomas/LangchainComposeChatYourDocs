<script lang="ts">

	import { conversation } from '$lib/conversationStore';
	import { goto } from '$app/navigation';
	import Prism from '$lib/components/PrismJS.svelte';
	export let collection: any;
	export let collections: any;
	export let index: any;
	let prompt: string = '';
	$: prompt = prompt;

	let mode = 0;
	$: mode = mode;
	let searching = false;
	const onKeyPress = async (e) => {
		if (e.charCode === 13) {
			let oldprompt = prompt;

			searching = true;
			prompt = '';
			const settings = {
				method: 'POST',
				headers: {
					Accept: 'application/json',
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					query: oldprompt,
					collection: collection,
					temperature: 0.5,
					index: index
				})
			};
			try {
				$conversation = [
					...$conversation,
					{
						bot: false,
						error: false,
						message: oldprompt,
						date: new Date().toLocaleDateString('nb-NO')
					}
				];
				const fetchResponse = await fetch(`/api/queryGPT`, settings);
				const data = await fetchResponse.json();

				$conversation = [
					...$conversation,
					{
						bot: true,
						error: false,
						message: data,
						date: new Date().toLocaleDateString('nb-NO')
					}
				];
				output = data;
				searching = false;
				return data;
			} catch (e) {
				$conversation = [
					...$conversation,
					{
						bot: true,
						error: true,
						message: 'Det skjedde en feil, prøv igjen senere...',
						date: new Date().toLocaleDateString('nb-NO')
					}
				];
				searching = false;
				prompt = oldprompt;
				return e;
			}
		}
	};

	$: selected = collection;



	import { fade, fly } from 'svelte/transition';
	import type { AnymatchFn } from 'vite';
	import { dataset_dev } from 'svelte/internal';
</script>

<div class="flex flex-col h-screen overflow-x-hidden">
	<div class="px-12 pt-12 pb-12 flex-grow overflow-x-hidden">
		<div class="chat chat-start overflow-x-hidden">
			<div class="chat-image avatar">
				<div class="w-16 mask mask-round">
					<img src="/profile-image.png" />
				</div>
			</div>

			<div class="chat-header">
				dagthomasBot
				<time class="text-xs opacity-50"
					>{new Date().toLocaleTimeString('nb-NO', { hour: 'numeric', minute: 'numeric' })}</time
				>
			</div>
			<div class="chat-bubble">
				<span>Hello there, I am <strong>dagthomasBot</strong>, what can I help you with? </span>
			</div>
		</div>
		{#each $conversation as item}
			{#if item?.bot}
				<div class="chat chat-start" in:fade={{ duration: 250 }} out:fade>
					<div class="chat-image  avatar">
						<div class="w-16 mask mask-round">
							<img src="/profile-image.png" />
						</div>
					</div>
					<div class="chat-header">
						dagthomasBot
						<time class="text-xs opacity-50"
							>{item?.data ||
								new Date().toLocaleTimeString('nb-NO', {
									hour: 'numeric',
									minute: 'numeric'
								})}</time
						>
					</div>
					<div class="chat-bubble">
						{#if item?.message?.output_text}
							{item?.message?.output_text}
						{:else}
							<Prism
								language="json"
								code={JSON.stringify(item?.message, null, 2)}
								header="Qdrant Search Results"
							/>
						{/if}
					</div>
				</div>
			{:else}
				<div class="chat chat-end" in:fly={{ x: 50, duration: 250 }} out:fade>
					<div class="chat-header">
						User
						<time class="text-xs opacity-50"
							>{item?.data ||
								new Date().toLocaleTimeString('nb-NO', {
									hour: 'numeric',
									minute: 'numeric'
								})}</time
						>
					</div>
					<div class="chat-bubble">
						{#if item?.message?.output_text}
							{item?.message?.output_text}
						{:else}
							{item?.message}
						{/if}
					</div>
				</div>
			{/if}
		{/each}
		{#if searching}
			<div class="chat chat-start">
				<div class="chat-image  avatar">
					<div class="w-16 mask mask-round">
						<img src="/profile-image.png" />
					</div>
				</div>

				<div class="chat-header">
					dagthomasBot
					<time class="text-xs opacity-50"
						>{new Date().toLocaleTimeString('nb-NO', { hour: 'numeric', minute: 'numeric' })}</time
					>
				</div>
				<div class="chat-bubble">
					<div
						role="status"
						class="absolute -translate-x-1/2 -translate-y-1/2 top-2/4 left-1/2 pl-2"
					>
						<svg
							aria-hidden="true"
							class="w-8 h-8 mr-2  animate-spin text-gray-600 fill-blue-600"
							viewBox="0 0 100 101"
							fill="none"
							xmlns="http://www.w3.org/2000/svg"
							><path
								d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
								fill="currentColor"
							/><path
								d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
								fill="currentFill"
							/></svg
						>
						<span class="sr-only">Writing answer...</span>
					</div>
				</div>
			</div>
		{/if}
	</div>
	<div class="navbar mb-4">
		<div class="flex-1 justify-center">
			<input
				type="text"
				placeholder="Ask a question"
				class="input input-bordered w-3/6 border-ll-s-1 border-2"
				bind:value={prompt}
				name="message"
				on:keypress={onKeyPress}
			/>
		</div>
	</div>
	<div class="absolute right-0 top-0 mt-8 mr-8">
		<h1 class="mb-2 ml-1 font-bold">Collections</h1>
		<select
			class="select select-bordered  max-w-xs select-sm"
			bind:value={selected}
			on:change={() => goto(selected)}
		>
			{#each collections as collection}
				<option value={collection.name}> {collection.name}</option>
			{/each}
		</select>
	</div>
</div>
